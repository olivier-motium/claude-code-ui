'use client';

/**
 * Form with Validation Example
 *
 * Demonstrates TanStack Form + Zod integration with:
 * - Schema-based validation
 * - Field arrays for dynamic inputs
 * - Async validation (username availability)
 * - TanStack Query mutation for submission
 * - Loading states and error display
 */

import { useForm } from '@tanstack/react-form';
import { zodValidator } from '@tanstack/zod-form-adapter';
import { z } from 'zod';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

// Validation Schema
const outcomeSchema = z.object({
  label: z.string().min(1, 'Label is required').max(50),
  probability: z.number().min(0, 'Min 0%').max(100, 'Max 100%'),
});

const marketSchema = z.object({
  name: z
    .string()
    .min(5, 'Name must be at least 5 characters')
    .max(100, 'Name must be at most 100 characters'),
  description: z
    .string()
    .min(20, 'Description must be at least 20 characters')
    .max(500, 'Description must be at most 500 characters'),
  category: z.enum(['politics', 'sports', 'crypto', 'entertainment', 'science'], {
    required_error: 'Please select a category',
  }),
  endDate: z.coerce.date().min(new Date(), 'End date must be in the future'),
  outcomes: z
    .array(outcomeSchema)
    .min(2, 'At least 2 outcomes required')
    .max(10, 'Maximum 10 outcomes')
    .refine(
      (outcomes) => {
        const total = outcomes.reduce((sum, o) => sum + o.probability, 0);
        return Math.abs(total - 100) < 0.01;
      },
      { message: 'Probabilities must sum to 100%' }
    ),
});

type MarketFormData = z.infer<typeof marketSchema>;

// Mock API
async function createMarket(data: MarketFormData): Promise<{ id: string }> {
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Simulate random failure (20% chance)
  if (Math.random() < 0.2) {
    throw new Error('Failed to create market. Please try again.');
  }

  return { id: `market-${Date.now()}` };
}

// Categories for dropdown
const categories = [
  { value: 'politics', label: 'Politics' },
  { value: 'sports', label: 'Sports' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'entertainment', label: 'Entertainment' },
  { value: 'science', label: 'Science' },
];

// Form Component
export function CreateMarketForm() {
  const queryClient = useQueryClient();
  const [submitResult, setSubmitResult] = useState<{ success: boolean; id?: string } | null>(null);

  // Mutation for form submission
  const createMutation = useMutation({
    mutationFn: createMarket,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['markets'] });
      setSubmitResult({ success: true, id: data.id });
    },
    onError: () => {
      setSubmitResult({ success: false });
    },
  });

  // Form instance
  const form = useForm({
    defaultValues: {
      name: '',
      description: '',
      category: '' as MarketFormData['category'],
      endDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 1 week from now
      outcomes: [
        { label: 'Yes', probability: 50 },
        { label: 'No', probability: 50 },
      ],
    } satisfies MarketFormData,
    validatorAdapter: zodValidator(),
    validators: {
      onChange: marketSchema,
    },
    onSubmit: async ({ value }) => {
      setSubmitResult(null);
      await createMutation.mutateAsync(value);
    },
  });

  // Calculate total probability
  const totalProbability = form.state.values.outcomes.reduce(
    (sum, o) => sum + (o.probability || 0),
    0
  );

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Create New Market</h1>

      {/* Success message */}
      {submitResult?.success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-800">Market created successfully!</p>
          <p className="text-green-600 text-sm">ID: {submitResult.id}</p>
        </div>
      )}

      {/* Error message */}
      {createMutation.isError && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{createMutation.error.message}</p>
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          form.handleSubmit();
        }}
        className="space-y-6"
      >
        {/* Name Field */}
        <form.Field name="name">
          {(field) => (
            <div>
              <label htmlFor={field.name} className="block text-sm font-medium mb-1">
                Market Name
              </label>
              <input
                id={field.name}
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
                className={`w-full px-3 py-2 border rounded-lg ${
                  field.state.meta.isTouched && field.state.meta.errors.length > 0
                    ? 'border-red-500'
                    : 'border-gray-300'
                }`}
                placeholder="e.g., Will Bitcoin reach $100k in 2024?"
              />
              {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
                <p className="mt-1 text-sm text-red-500">{field.state.meta.errors[0]}</p>
              )}
            </div>
          )}
        </form.Field>

        {/* Description Field */}
        <form.Field name="description">
          {(field) => (
            <div>
              <label htmlFor={field.name} className="block text-sm font-medium mb-1">
                Description
              </label>
              <textarea
                id={field.name}
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
                rows={3}
                className={`w-full px-3 py-2 border rounded-lg ${
                  field.state.meta.isTouched && field.state.meta.errors.length > 0
                    ? 'border-red-500'
                    : 'border-gray-300'
                }`}
                placeholder="Provide detailed resolution criteria..."
              />
              {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
                <p className="mt-1 text-sm text-red-500">{field.state.meta.errors[0]}</p>
              )}
            </div>
          )}
        </form.Field>

        {/* Category Field */}
        <form.Field name="category">
          {(field) => (
            <div>
              <label htmlFor={field.name} className="block text-sm font-medium mb-1">
                Category
              </label>
              <select
                id={field.name}
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value as MarketFormData['category'])}
                onBlur={field.handleBlur}
                className={`w-full px-3 py-2 border rounded-lg ${
                  field.state.meta.isTouched && field.state.meta.errors.length > 0
                    ? 'border-red-500'
                    : 'border-gray-300'
                }`}
              >
                <option value="">Select a category</option>
                {categories.map((cat) => (
                  <option key={cat.value} value={cat.value}>
                    {cat.label}
                  </option>
                ))}
              </select>
              {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
                <p className="mt-1 text-sm text-red-500">{field.state.meta.errors[0]}</p>
              )}
            </div>
          )}
        </form.Field>

        {/* End Date Field */}
        <form.Field name="endDate">
          {(field) => (
            <div>
              <label htmlFor={field.name} className="block text-sm font-medium mb-1">
                End Date
              </label>
              <input
                id={field.name}
                type="datetime-local"
                value={field.state.value.toISOString().slice(0, 16)}
                onChange={(e) => field.handleChange(new Date(e.target.value))}
                onBlur={field.handleBlur}
                className={`w-full px-3 py-2 border rounded-lg ${
                  field.state.meta.isTouched && field.state.meta.errors.length > 0
                    ? 'border-red-500'
                    : 'border-gray-300'
                }`}
              />
              {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
                <p className="mt-1 text-sm text-red-500">{field.state.meta.errors[0]}</p>
              )}
            </div>
          )}
        </form.Field>

        {/* Outcomes Field Array */}
        <form.Field name="outcomes" mode="array">
          {(field) => (
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-medium">
                  Outcomes
                  <span className={`ml-2 ${Math.abs(totalProbability - 100) < 0.01 ? 'text-green-600' : 'text-red-500'}`}>
                    (Total: {totalProbability.toFixed(0)}%)
                  </span>
                </label>
                <button
                  type="button"
                  onClick={() => field.pushValue({ label: '', probability: 0 })}
                  disabled={field.state.value.length >= 10}
                  className="text-sm px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  Add Outcome
                </button>
              </div>

              <div className="space-y-3">
                {field.state.value.map((_, index) => (
                  <div key={index} className="flex gap-3 items-start">
                    <form.Field name={`outcomes[${index}].label`}>
                      {(subField) => (
                        <div className="flex-1">
                          <input
                            value={subField.state.value}
                            onChange={(e) => subField.handleChange(e.target.value)}
                            onBlur={subField.handleBlur}
                            placeholder="Outcome label"
                            className={`w-full px-3 py-2 border rounded-lg ${
                              subField.state.meta.isTouched && subField.state.meta.errors.length > 0
                                ? 'border-red-500'
                                : 'border-gray-300'
                            }`}
                          />
                        </div>
                      )}
                    </form.Field>

                    <form.Field name={`outcomes[${index}].probability`}>
                      {(subField) => (
                        <div className="w-24">
                          <div className="relative">
                            <input
                              type="number"
                              value={subField.state.value}
                              onChange={(e) => subField.handleChange(Number(e.target.value))}
                              onBlur={subField.handleBlur}
                              min={0}
                              max={100}
                              className={`w-full px-3 py-2 border rounded-lg pr-8 ${
                                subField.state.meta.isTouched && subField.state.meta.errors.length > 0
                                  ? 'border-red-500'
                                  : 'border-gray-300'
                              }`}
                            />
                            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
                              %
                            </span>
                          </div>
                        </div>
                      )}
                    </form.Field>

                    {field.state.value.length > 2 && (
                      <button
                        type="button"
                        onClick={() => field.removeValue(index)}
                        className="px-3 py-2 text-red-600 hover:bg-red-50 rounded"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {field.state.meta.errors.length > 0 && (
                <p className="mt-2 text-sm text-red-500">{field.state.meta.errors[0]}</p>
              )}
            </div>
          )}
        </form.Field>

        {/* Submit Button */}
        <div className="flex gap-4">
          <button
            type="submit"
            disabled={!form.state.canSubmit || form.state.isSubmitting}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {form.state.isSubmitting ? 'Creating...' : 'Create Market'}
          </button>

          <button
            type="button"
            onClick={() => form.reset()}
            disabled={!form.state.isDirty}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  );
}

export default CreateMarketForm;
